const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {"start":"_app/immutable/entry/start.CFha25r8.js","app":"_app/immutable/entry/app.DLZYTOsA.js","imports":["_app/immutable/entry/start.CFha25r8.js","_app/immutable/chunks/client.zKyAK13j.js","_app/immutable/entry/app.DLZYTOsA.js","_app/immutable/chunks/preload-helper.DpQnamwV.js"],"stylesheets":[],"fonts":[],"uses_env_dynamic_public":false},
		nodes: [
			__memo(() => import('./chunks/0-BOe8I1Po.js')),
			__memo(() => import('./chunks/1-BMM1miYc.js')),
			__memo(() => import('./chunks/2-DzJuCZmo.js').then(function (n) { return n.aE; }))
		],
		routes: [
			{
				id: "/[...catchall]",
				pattern: /^(?:\/(.*))?\/?$/,
				params: [{"name":"catchall","optional":false,"rest":true,"chained":true}],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();

const prerendered = new Set([]);

const base = "";

export { base, manifest, prerendered };
//# sourceMappingURL=manifest.js.map
